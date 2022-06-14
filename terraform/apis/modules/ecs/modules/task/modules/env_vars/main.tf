variable "env_vars" {
  description = "Environment variables to pass to the container"
  type        = map(string)
}

output "env_vars_string" {
  value = jsonencode([
    for key in keys(var.env_vars) :
    {
      name  = key
      value = lookup(var.env_vars, key)
    }
  ])
}
