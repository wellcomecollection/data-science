resource "aws_s3_bucket" "miro_images_feature_vectors" {
  bucket = "miro-images-feature-vectors"
  acl    = "private"

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_s3_bucket" "model_core_data" {
  bucket = "model-core-data"
  acl    = "private"

  lifecycle {
    prevent_destroy = true
  }
}

# policies
resource "aws_s3_bucket_policy" "model_core_data_policy" {
  bucket = aws_s3_bucket.model_core_data.bucket
  policy = data.aws_iam_policy_document.s3_read.json
}

data "aws_iam_policy_document" "s3_read" {
  statement {
    actions = ["s3:*"]

    resources = [
      aws_s3_bucket.model_core_data.arn,
      "${aws_s3_bucket.model_core_data.arn}/*",
    ]

    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::964279923020:role/datascience_ec2"]
    }
  }
}
