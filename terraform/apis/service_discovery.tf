resource "aws_service_discovery_private_dns_namespace" "namespace" {
  name = "datascience"
  vpc  = "${local.vpc_id}"
}
