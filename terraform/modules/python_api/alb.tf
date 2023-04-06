resource "aws_alb" "python_api" {
  name            = var.name
  subnets         = var.public_subnets
  security_groups = [aws_security_group.lb.id]

  load_balancer_type = "application"
  internal           = false
}

resource "aws_alb_listener" "https" {
  load_balancer_arn = aws_alb.python_api.arn
  port              = "443"
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-2016-08"
  certificate_arn   = module.cert.arn

  default_action {
    type             = "forward"
    target_group_arn = module.service.target_group_arn
  }
}

resource "aws_alb_listener" "http" {
  load_balancer_arn = aws_alb.python_api.id
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type = "redirect"

    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }
}
