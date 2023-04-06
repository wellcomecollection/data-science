module "cert" {
  source = "github.com/wellcomecollection/terraform-aws-acm-certificate?ref=v1.0.0"

  domain_name = local.domain_name

  zone_id = data.aws_route53_zone.dotorg.zone_id

  providers = {
    aws.dns = aws.dns
  }
}

data "aws_route53_zone" "dotorg" {
  provider = aws.dns

  name = "wellcomecollection.org."
}
