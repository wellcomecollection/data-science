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
        line-height: 1.45em;
      }

      main {
        background: white;
        padding: 1em;
        border-radius: 10px;
      }

      main h1 {
        margin-top: 1em;
      }

      pre {
        background: #e8e8e8;
        overflow: scroll;
        padding: 10px;
        font-size: 1.15em;
        line-height: 1.4em;
        border-radius: 3px;
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
        You can access this API at URL <strong><a href="http://${domain_name}">http://${domain_name}</a></strong>.
      </p>

      <p>

        You can view application logs <strong><a href="https://logging.wellcomecollection.org/app/discover#/?_g=(filters:!(),refreshInterval:(pause:!t,value:0),time:(from:now-15m,to:now))&_a=(columns:!(log),filters:!(('$state':(store:appState),meta:(alias:!n,disabled:!f,index:cb5ba262-ec15-46e3-a4c5-5668d65fe21f,key:ecs_cluster,negate:!f,params:(query:${cluster_name}),type:phrase),query:(match_phrase:(ecs_cluster:${cluster_name})))),index:cb5ba262-ec15-46e3-a4c5-5668d65fe21f,interval:auto,query:(language:kuery,query:''),sort:!(!('@timestamp',desc)))">in the logging cluster</a></strong>.
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
  --force-new-deployment</code></pre>

          <details>
            <summary>If the app starts but struggles to stay up</summary>
            <p>You may see tasks start, the API serve a few requests, then get shut down by ECS. If you see the following error in the app logs:</p>

            <pre><code>"GET / HTTP/1.1" 404 22 "-" "ELB-HealthChecker/2.0"</code></pre>

            <p>then you app is getting shut down by the load balancer, because it thinks the app is unhealthy. You need to add an HTTP 200 response for the <code>/</code> endpoint in your app, then redeploy.</p>
          </details>
      </ol>
    </main>
  </body>
</html>
