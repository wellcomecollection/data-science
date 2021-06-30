resource "aws_lb" "network_load_balancer" {
  name               = "datascience-apis-nlb"
  internal           = true
  load_balancer_type = "network"
  subnets            = local.private_subnets
}
