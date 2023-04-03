variable "name" {
  type = string
}

variable "subnets" {
  type = list(string)
}

variable "vpc_id" {
  type = string
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
  type    = number
}

variable "memory" {
  type    = number
}

variable "container_port" {
  type = number
}

variable "nginx_container_config" {
  type = object({
    image_name    = string
    container_tag = string
  })

  # This has an increased max request body size, and increased proxy buffer sizes
  # It's copied from the front-end repo
  default = {
    image_name    = "uk.ac.wellcome/nginx_frontend"
    container_tag = "9b95057b716a60f9891f77111b0bd524b85839aa"
  }
}
