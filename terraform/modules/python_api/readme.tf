resource "local_file" "readme" {
  content = templatefile("${path.module}/README.html.tpl", {
    name         = var.name
    ecr_repo_url = aws_ecr_repository.api.repository_url
    cluster_name = aws_ecs_cluster.cluster.name
    service_name = module.service.name
    domain_name  = local.domain_name
  })

  filename = "../api.README.html"
}
