resource "local_file" "readme" {
  content  = templatefile("${path.module}/README.html.tpl", {
    name          = var.name
    ecr_repo_url = aws_ecr_repository.api.repository_url
  })

  filename = "../api.README.html"
}
