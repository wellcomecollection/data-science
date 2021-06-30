variable "subnets" {
  type = list(string)
}

variable "cluster_id" {}

variable "namespace" {}
variable "namespace_id" {}

variable "vpc_id" {}

variable "container_image" {}

variable "container_port" {
  default = "80"
}

variable "nginx_container_image" {}
variable "nginx_container_port" {}

variable "security_group_ids" {
  type = list(string)
}

variable "env_vars_length" {}

variable "env_vars" {
  type = map(string)
}

variable "lb_arn" {}
variable "lb_dns_name" {}
variable "listener_port" {}

variable "api_gateway_rest_api_id" {}
variable "api_gateway_root_resource_id" {}
variable "api_gateway_vpc_link_id" {}

variable "cpu" {
  default = 1024
}

variable "memory" {
  default = 2048
}

variable "sidecar_cpu" {
  default = "512"
}

variable "sidecar_memory" {
  default = "1024"
}

variable "app_cpu" {
  default = "512"
}

variable "app_memory" {
  default = "1024"
}

variable "aws_region" {
  default = "eu-west-1"
}

variable "launch_type" {
  default = "FARGATE"
}

variable "task_desired_count" {
  default = "3"
}

variable "assumable_roles" {
  type    = list(string)
  default = []
}
