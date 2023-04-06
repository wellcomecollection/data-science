resource "aws_alb" "python_api" {
  name            = var.name
  subnets         = var.public_subnets
  security_groups = [aws_security_group.lb.id]

  load_balancer_type = "application"
  internal           = false
}

locals {
  domain_name = aws_alb.python_api.dns_name
}

resource "aws_alb_listener" "http" {
  load_balancer_arn = aws_alb.python_api.id
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = module.service.target_group_arn
  }
}
