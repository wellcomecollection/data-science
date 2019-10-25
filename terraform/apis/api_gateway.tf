resource "aws_api_gateway_vpc_link" "apis" {
  name        = "apis"
  description = "APIs"
  target_arns = ["${module.nlb.arn}"]
}

resource "aws_api_gateway_rest_api" "apis" {
  name        = "apis"
  description = "Gateway for the all the datascience APIs"
}

resource "aws_api_gateway_deployment" "apis" {
  # TODO: Get the deployment deps in here
  rest_api_id = "${aws_api_gateway_rest_api.apis.id}"
  stage_name  = "test"
}
