module "palette_api" {
  source = "./modules/service/api"

  namespace = "palette-api"

  container_image = "${local.palette_api_container_image}"
  container_port  = "${local.palette_api_container_port}"

  namespace_id = "${aws_service_discovery_private_dns_namespace.namespace.id}"

  cluster_id = "${aws_ecs_cluster.cluster.name}"

  vpc_id = "${local.vpc_id}"

  security_group_ids = [
    "${aws_security_group.service_egress_security_group.id}",
    "${aws_security_group.service_lb_ingress_security_group.id}",
  ]

  assumable_roles = ["${local.miro_read_role}"]

  subnets                      = ["${local.private_subnets}"]
  nginx_container_port         = "${local.nginx_listener_port}"
  nginx_container_image        = "${local.nginx_container_image}"
  env_vars                     = {}
  env_vars_length              = 0
  lb_arn                       = "${module.nlb.arn}"
  lb_dns_name                  = "${module.nlb.dns_name}"
  api_gateway_rest_api_id      = "${aws_api_gateway_rest_api.apis.id}"
  api_gateway_root_resource_id = "${aws_api_gateway_rest_api.apis.root_resource_id}"
  api_gateway_vpc_link_id      = "${aws_api_gateway_vpc_link.apis.id}"
  listener_port                = "${local.palette_api_listener_port}"
  cpu                          = 2048
  memory                       = 4096
  sidecar_cpu                  = 1024
  sidecar_memory               = 2048
  app_cpu                      = 1024
  app_memory                   = 2048
  task_desired_count           = "${local.palette_api_task_desired_count}"
}
