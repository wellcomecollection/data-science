variable "aws_region" {}

variable "task_name" {}

variable "log_group_prefix" {
  description = "Cloudwatch log group name prefix"
  default     = "ecs"
}

# App

variable "app_container_image" {}

variable "app_port_mappings_string" {
  default = "[]"
}

variable "app_cpu" {}
variable "app_memory" {}

variable "app_mount_points" {
  type    = list(string)
  default = []
}

variable "app_env_vars" {
  description = "Environment variables to pass to the container"
  type        = map(string)
  default     = {}
}

# Sidecar

variable "sidecar_container_image" {}

variable "sidecar_port_mappings_string" {
  default = "[]"
}

variable "sidecar_cpu" {}
variable "sidecar_memory" {}

variable "sidecar_mount_points" {
  type    = list(string)
  default = []
}

variable "sidecar_env_vars" {
  description = "Environment variables to pass to the container"
  type        = map(string)
  default     = {}
}
