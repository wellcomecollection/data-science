locals {
  domain_name = "${var.name}.datascience.wellcomecollection.org"
}

resource "aws_route53_record" "python_api" {
  provider = aws.dns

  name    = local.domain_name
  zone_id = data.aws_route53_zone.dotorg.id
  type    = "A"

  alias {
    name                   = aws_alb.python_api.dns_name
    zone_id                = aws_alb.python_api.zone_id
    evaluate_target_health = false
  }
}