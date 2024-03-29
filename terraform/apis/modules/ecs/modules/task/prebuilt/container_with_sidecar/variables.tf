variable "task_name" {}

variable "log_group_prefix" {
  description = "Cloudwatch log group name prefix"
  default     = "ecs"
}

variable "cpu" {}
variable "memory" {}

variable "app_container_image" {}
variable "app_container_port" {}
variable "app_cpu" {}
variable "app_memory" {}

variable "app_env_vars" {
  description = "Environment variables to pass to the container"
  type        = map(string)
  default     = {}
}

variable "sidecar_container_image" {}
variable "sidecar_container_port" {}
variable "sidecar_cpu" {}
variable "sidecar_memory" {}

variable "sidecar_env_vars" {
  description = "Environment variables to pass to the container"
  type        = map(string)
  default     = {}
}

variable "aws_region" {}

variable "launch_types" {
  type    = list(string)
  default = ["FARGATE", "EC2"]
}
