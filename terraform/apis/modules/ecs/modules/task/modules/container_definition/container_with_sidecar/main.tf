locals {
  app_mount_points     = jsonencode(var.app_mount_points)
  sidecar_mount_points = jsonencode(var.app_mount_points)

  app_log_group_name     = var.task_name
  sidecar_log_group_name = "sidecar_${var.task_name}"

  app_container_name     = "app"
  sidecar_container_name = "sidecar"

  task_definition_template_path = "${path.module}/task_definition.json.template"
  task_definition_template_vars = {
    log_group_region = var.aws_region
    log_group_prefix = var.log_group_prefix

    # App vars
    app_log_group_name = aws_cloudwatch_log_group.app.name

    app_container_image = var.app_container_image
    app_container_name  = local.app_container_name
    app_port_mappings   = var.app_port_mappings_string

    app_environment_vars = module.app_env_vars.env_vars_string

    app_cpu    = var.app_cpu
    app_memory = var.app_memory

    app_mount_points = local.app_mount_points

    # Sidecar vars
    sidecar_log_group_name = aws_cloudwatch_log_group.sidecar.name

    sidecar_container_image = var.sidecar_container_image
    sidecar_container_name  = local.sidecar_container_name

    sidecar_port_mappings = var.sidecar_port_mappings_string

    sidecar_environment_vars = module.sidecar_env_vars.env_vars_string

    sidecar_cpu    = var.sidecar_cpu
    sidecar_memory = var.sidecar_memory

    sidecar_mount_points = local.sidecar_mount_points
  }
}

# Sidecar

resource "aws_cloudwatch_log_group" "sidecar" {
  name = "ecs/sidecar_${var.task_name}"

  retention_in_days = 7
}

module "sidecar_env_vars" {
  source          = "../../env_vars"
  env_vars        = var.sidecar_env_vars
  env_vars_length = var.sidecar_env_vars_length
}

# App

resource "aws_cloudwatch_log_group" "app" {
  name = "ecs/${var.task_name}"

  retention_in_days = 7
}

module "app_env_vars" {
  source          = "../../env_vars"
  env_vars        = var.app_env_vars
  env_vars_length = var.app_env_vars_length
}
