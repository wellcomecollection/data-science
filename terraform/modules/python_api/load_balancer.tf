resource "aws_security_group" "allow_all_inbound" {
  name        = "${var.name}-inbound"
  description = "Allow traffic from anywhere on the Internet"
  vpc_id      = var.vpc_id

  ingress {
    from_port = 0
    to_port   = 0
    protocol  = "-1"
    self      = true
  }
}

resource "aws_alb" "alb" {
  name            = var.name
  subnets         = var.subnets
  security_groups = [aws_security_group.allow_all_inbound.id]
}

resource "aws_alb_target_group" "ecs_service_default" {
  name = "${var.name}-default"
  port = 80
  protocol = "HTTP"
  vpc_id = var.vpc_id
}

resource "aws_alb_listener" "http" {
  load_balancer_arn = aws_alb.alb.id
  port = "80"
  protocol = "HTTP"

  default_action {
    target_group_arn = aws_alb_target_group.ecs_service_default.arn
    type = "forward"
  }
}

resource "aws_lb_listener_rule" "http" {
  listener_arn = aws_alb_listener.http.arn
  priority     = 100

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.http.arn
  }

  condition {
    path_pattern {
      values = ["/*"]
    }
  }
}
