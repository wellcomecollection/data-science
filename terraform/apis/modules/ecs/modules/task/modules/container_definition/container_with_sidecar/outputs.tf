output "rendered" {
  value = templatefile(
    local.task_definition_template_path,
    local.task_definition_template_vars
  )
}

output "app_container_name" {
  value = local.app_container_name
}

output "sidecar_container_name" {
  value = local.sidecar_container_name
}
