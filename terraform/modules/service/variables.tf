variable "subnets" {
  type = list(string)
}

variable "cluster_arn" {}

variable "service_name" {}

variable "vpc_id" {}

variable "container_image" {}
variable "container_port" {
  type = number
}

variable "security_group_ids" {
  type = list(string)
}

variable "env_vars" {
  type    = map(string)
  default = {}
}

variable "secret_env_vars" {
  type    = map(string)
  default = {}
}

variable "cpu" {
  type = number
}

variable "memory" {
  type = number
}

variable "desired_task_count" {
  default = 3
}
