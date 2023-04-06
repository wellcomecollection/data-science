resource "aws_ecs_cluster" "cluster" {
  name = "data-science-${var.name}"
}

module "service" {
  source = "../service"

  service_name = var.name

  vpc_id  = var.vpc_id
  subnets = var.private_subnets

  container_image = "${aws_ecr_repository.api.repository_url}:latest"
  container_port  = var.container_port

  env_vars        = var.env_vars
  secret_env_vars = var.secret_env_vars

  cpu    = var.cpu
  memory = var.memory

  security_group_ids = [aws_security_group.lb_ingress.id, aws_security_group.egress.id]

  cluster_arn = aws_ecs_cluster.cluster.arn
}