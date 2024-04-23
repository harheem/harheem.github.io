---
layout: page
title: "How to use psycopg2 in an AWS Lambda"
subtitle: "Python version 3.12, Docker"
date:   2024-04-08 14:31:24 +0530
categories: Dev
author: "Harheem Kim"
toc: true
---

# How to use psycopg2 with Python version 3.12 in an AWS Lambda

> This article describes how to use psycopg2 with Python version 3.12 in an AWS Lambda environment. The process involves adjusting and applying methods verified in Python 3.9 to suit the Python 3.12 environment.
> 

- **`psycopg2`** is a Python library for connecting to PostgreSQL databases. PostgreSQL is an open-source relational database system widely used in various projects and applications. psycopg2 is a tool that facilitates interaction with PostgreSQL.
- **`AWS Lambda`** is a service that allows code execution without a server, charging only for the computing resources used. This service offers several advantages, including automatic scalability and event-based execution.

## **Preparing psycopg2 in Python 3.9 (Using Docker)**

---

[Unable to import module 'testdb': No module named 'psycopg2._psycopg' · aws aws-cdk · Discussion #28339](https://github.com/aws/aws-cdk/discussions/28339)

I realized that the questioner was facing the same issue I had experienced. One of the answers suggested using Docker, and I decided to test this method out.

First, create a working directory and record psycopg2-binary in the requirements.txt file.

```python
mkdir lambda
cd lambda
echo 'psycopg2-binary' > requirements.txt
```

Next, use Docker to install the necessary libraries in the Python 3.9 environment and create a psycopg2.zip file.

```python
docker run -v "$PWD":/var/task "amazon/aws-sam-cli-build-image-python3.9" /bin/sh -c "pip install -r requirements.txt -t python/lib/python3.9/site-packages/; exit"
zip -r psycopg2.zip python
```

By this method, the created psycopg2.zip file can be registered as a Lambda layer, allowing you to import psycopg2. This is a safe method as it involves downloading psycopg2 in AWS Linux.

You can learn more about registering a Lambda layer in [this article](https://docs.aws.amazon.com/ko_kr/lambda/latest/dg/adding-layers.html).

## **Preparing psycopg2 in Python 3.12 (Using Docker)**

---

The same process was then applied in the Python 3.12 environment using the following commands.

```python
echo 'psycopg2-binary' > requirements.txt
docker run -v "$PWD":/var/task "public.ecr.aws/sam/build-python3.12:latest" /bin/sh -c "pip install -r requirements.txt -t python/lib/python3.12/site-packages/; exit"
zip -r psycogpg2.zip python
```

```python

Python 3.12.2 (main, Mar 15 2024, 11:09:09) [GCC 11.4.1 20230605 (Red Hat 11.4.1-2)] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import sys
>>> sys.path.insert(0, "/var/task/python/lib/python3.12/site-packages/")
>>> import psycopg2
>>> psycopg2.__version__
'2.9.9 (dt dec pq3 ext lo64)'
```

In the Docker container environment, I confirmed that psycopg2 was successfully imported. 

After registering it as a Lambda layer using the same method, I executed the code.

![Untitled](https://www.notion.so/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F2a330106-7d16-49d5-9057-343dfb0cb92c%2F6a14bf8d-a450-46db-a868-0e028a24fd54%2FUntitled.png?table=block&id=d213fc2c-e291-4fa4-8820-38164aec72eb&spaceId=2a330106-7d16-49d5-9057-343dfb0cb92c&width=2000&userId=93a922d0-0a24-445b-bddf-8a085b93d655&cache=v2)

![Untitled](https://www.notion.so/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F2a330106-7d16-49d5-9057-343dfb0cb92c%2Fd92ffa5e-f133-4f3a-978b-15d654b84559%2FUntitled.png?table=block&id=92a8d8b9-45dc-444e-9b1d-928cbafc7eaa&spaceId=2a330106-7d16-49d5-9057-343dfb0cb92c&width=2000&userId=93a922d0-0a24-445b-bddf-8a085b93d655&cache=v2)

As a result, the psycopg2 version was correctly displayed in the Python 3.12 environment. The image used was [public.ecr.aws/sam/build-python3.12:latest](http://public.ecr.aws/sam/build-python3.12:latest), and the Lambda function was set up for Python 3.12 arm64. If a different architecture is needed, you can find the desired image at [this link](https://gallery.ecr.aws/sam/build-python3.12).

The registered ARN is as follows. Registering it as a layer is convenient as it can be easily used in other functions.

```
arn:aws:lambda:ap-northeast-2:550316102722:layer:psycopg2-binary-arm64-312:1
```