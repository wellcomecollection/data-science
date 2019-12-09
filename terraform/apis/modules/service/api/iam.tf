locals {
  assume_policy_count = "${length(var.assumable_roles) > 0 ? 1 : 0}"
}

data "aws_iam_policy_document" "allow_assume_roles" {
  statement {
    actions = [
      "sts:AssumeRole",
    ]

    resources = ["${var.assumable_roles}"]
  }
}

data "aws_iam_policy_document" "allow_read_core_data" {
  statement {
    actions = [
      "s3:ListBucket",
      "s3:GetBucketLocation",
    ]

    resources = ["${local.core_data_bucket}"]
  }

  statement {
    actions = [
      "s3:GetObject",
    ]

    resources = ["${local.core_data_bucket}/*"]
  }
}

resource "aws_iam_policy" "allow_assume_roles" {
  count       = "${local.assume_policy_count}"
  name        = "${var.namespace}-allow-assume-roles"
  description = "Allows the specified roles to be assumed"
  policy      = "${data.aws_iam_policy_document.allow_assume_roles.json}"
}

resource "aws_iam_policy" "allow_read_core_data" {
  name        = "${var.namespace}-allow-read-core-data"
  description = "Allows reading the core data bucket"
  policy      = "${data.aws_iam_policy_document.allow_read_core_data.json}"
}

resource "aws_iam_role_policy_attachment" "attach_allow_assume_roles" {
  count      = "${local.assume_policy_count}"
  role       = "${module.task.task_role_name}"
  policy_arn = "${aws_iam_policy.allow_assume_roles.arn}"
}

resource "aws_iam_role_policy_attachment" "attach_allow_read_core_data" {
  role       = "${module.task.task_role_name}"
  policy_arn = "${aws_iam_policy.allow_read_core_data.arn}"
}
