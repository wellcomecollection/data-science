<html>
  <head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">

    <title>README for ${name} in AWS</title>

    <style>
      body {
        max-width: 800px;
        margin-left:  auto;
        margin-right: auto;
        background: #fff9e6;
        font-family: -apple-system, sans-serif;
      }

      main {
        background: white;
        padding: 1em;
        border-radius: 10px;
      }

      pre {
        background: #e8e8e8;
        overflow: scroll;
        padding: 10px;
        font-size: 1.15em;
        line-height: 1.4em;
      }

      details {
        border-left: 4px solid #ff6f59;
        padding: 10px;
        background: rgba(255, 149, 133, 0.2);
      }

      details p:last-child {
        margin-bottom: 0;
      }
    </style>
  </head>
  <body>
    <main>
      <h1>README for ${name}</h1>

      <p>
        You can access this API at URL <strong><a href="https://${domain_name}">${domain_name}</a></strong>.
      </p>

      <h2>Deploying a new version of the API</h2>

      <ol>
        <li><p>Build a copy of the Docker image from the API:</p>

          <pre><code>docker build --tag ${name} […]</code></pre></li>

        <li>
          <p>Log in to ECR.  This will allow you to push Docker images.</p>

          <pre><code>eval $(AWS_PROFILE=data-dev aws ecr get-login --no-include-email)</code></pre>

          <details>
            <summary>If you get an error "The config profile could not be found"</summary>
            <p>If you get an error like:</p>

            <pre><code>The config profile (data-dev) could not be found</code></pre>

            <p>then you may need to configure your AWS roles.  It's probably enough to add this to <code>~/.aws/credentials</code>:</p>

            <pre><code>[data-dev]
  source_profile=default
  role_arn=arn:aws:iam::964279923020:role/data-developer
  region=eu-west-1</code></pre>

            <p>If that doesn't work, ask for help in Slack.</p>
          </details>
        </li>

        <li>
          <p>Tag the Docker image you built in step 1 with the name of the ECR repository:</p>

          <pre><code>docker tag ${name} ${ecr_repo_url}:latest</code></pre>
        </li>

        <li>
          <p>Push the tagged image to the ECR repository:</p>

          <pre><code>docker push ${ecr_repo_url}:latest</code></pre>
        </li>

        <li>
          <p>Tell ECS to redeploy the API using your new images by running the following command:</p>

          <pre><code>AWS_PROFILE=data-dev aws ecs update-service \
  --service ${service_name} \
  --cluster ${cluster_name} \
  --force-new-deployment
      </ol>
    </main>
  </body>
</html>
