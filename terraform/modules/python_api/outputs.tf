output "domain_name" {
  value = local.domain_name
}

output "readme_path" {
  value = local_file.readme.filename
}
