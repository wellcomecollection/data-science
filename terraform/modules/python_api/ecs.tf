resource "aws_ecs_cluster" "cluster" {
  name = "data-science-${var.name}"
}

module "log_router_container" {
  source    = "git::github.com/wellcomecollection/terraform-aws-ecs-service.git//modules/firelens?ref=v3.13.1"
  namespace = var.name
}

module "log_router_container_secrets_permissions" {
  source    = "git::github.com/wellcomecollection/terraform-aws-ecs-service.git//modules/secrets?ref=v3.13.1"
  secrets   = module.log_router_container.shared_secrets_logging
  role_name = module.task_definition.task_execution_role_name
}

module "app_container" {
  source = "git::github.com/wellcomecollection/terraform-aws-ecs-service.git//modules/container_definition?ref=v3.13.1"
  name   = "app"

  image = "${aws_ecr_repository.api.repository_url}:latest"

  environment = var.env_vars
  secrets     = var.secret_env_vars

  log_configuration = module.log_router_container.container_log_configuration
}

module "app_container_secrets_permissions" {
  source    = "git::github.com/wellcomecollection/terraform-aws-ecs-service.git//modules/secrets?ref=v3.13.1"
  secrets   = var.secret_env_vars
  role_name = module.task_definition.task_execution_role_name
}

module "nginx_container" {
  source = "git::github.com/wellcomecollection/terraform-aws-ecs-service.git//modules/nginx/frontend?ref=v3.13.1"

  forward_port      = var.container_port
  log_configuration = module.log_router_container.container_log_configuration

  container_tag = var.nginx_container_config["container_tag"]
  image_name    = var.nginx_container_config["image_name"]
}

module "task_definition" {
  source = "git::github.com/wellcomecollection/terraform-aws-ecs-service.git//modules/task_definition?ref=v3.13.1"

  cpu    = var.cpu
  memory = var.memory

  container_definitions = [
    module.log_router_container.container_definition,
    module.nginx_container.container_definition,
    module.app_container.container_definition
  ]

  task_name = var.name
}

module "service" {
  source = "git::github.com/wellcomecollection/terraform-aws-ecs-service.git//modules/service?ref=v3.13.0"

  cluster_arn  = aws_ecs_cluster.cluster.arn
  service_name = var.name

  task_definition_arn = module.task_definition.arn

  subnets            = var.subnets
  security_group_ids = [aws_security_group.allow_all_ingress.id, aws_security_group.allow_all_egress.id]

  desired_task_count = 3
  use_fargate_spot   = true

  target_group_arn = aws_lb_target_group.http.arn

  container_name = "nginx"
  container_port = module.nginx_container.container_port

  propagate_tags = "SERVICE"
}

resource "aws_lb_target_group" "http" {
  name                 = "${var.name}-http"
  port                 = module.nginx_container.container_port
  protocol             = "HTTP"
  vpc_id               = var.vpc_id
  deregistration_delay = 10
  target_type          = "ip"

  lifecycle {
    create_before_destroy = true
  }
}
