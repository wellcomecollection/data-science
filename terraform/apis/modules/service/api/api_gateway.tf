resource "aws_api_gateway_resource" "api" {
  rest_api_id = "${var.api_gateway_rest_api_id}"
  parent_id   = "${var.api_gateway_root_resource_id}"
  path_part   = "${var.namespace}"
}

resource "aws_api_gateway_resource" "api_proxy" {
  rest_api_id = "${var.api_gateway_rest_api_id}"
  parent_id   = "${aws_api_gateway_resource.api.id}"
  path_part   = "{proxy+}"
}

resource "aws_api_gateway_method" "api" {
  rest_api_id   = "${var.api_gateway_rest_api_id}"
  resource_id   = "${aws_api_gateway_resource.api.id}"
  http_method   = "GET"
  authorization = "NONE"
}

resource "aws_api_gateway_method" "api_proxy" {
  rest_api_id   = "${var.api_gateway_rest_api_id}"
  resource_id   = "${aws_api_gateway_resource.api_proxy.id}"
  http_method   = "GET"
  authorization = "NONE"

  request_parameters = {
    "method.request.path.proxy" = true
  }
}

resource "aws_api_gateway_integration" "api" {
  rest_api_id = "${var.api_gateway_rest_api_id}"
  resource_id = "${aws_api_gateway_method.api.resource_id}"
  http_method = "${aws_api_gateway_method.api.http_method}"

  integration_http_method = "GET"
  type                    = "HTTP_PROXY"
  uri                     = "http://${var.lb_dns_name}:${var.listener_port}"

  connection_type = "VPC_LINK"
  connection_id   = "${var.api_gateway_vpc_link_id}"
}

resource "aws_api_gateway_integration" "api_proxy" {
  rest_api_id = "${var.api_gateway_rest_api_id}"
  resource_id = "${aws_api_gateway_method.api_proxy.resource_id}"
  http_method = "${aws_api_gateway_method.api_proxy.http_method}"

  integration_http_method = "GET"
  type                    = "HTTP_PROXY"
  uri                     = "http://${var.lb_dns_name}:${var.listener_port}/{proxy}"

  connection_type = "VPC_LINK"
  connection_id   = "${var.api_gateway_vpc_link_id}"

  request_parameters = {
    "integration.request.path.proxy" = "method.request.path.proxy"
  }
}
