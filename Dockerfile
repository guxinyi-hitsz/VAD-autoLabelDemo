FROM python:3.7.7 as builder

COPY requirements.txt /srv/requirements.txt

RUN pip download -r /srv/requirements.txt -d /srv/pkgs_cache --index-url=https://mirrors.aliyun.com/pypi/simple/
RUN pip install -r /srv/requirements.txt --target /srv/pkgs --find-links=/srv/pkgs_cache --no-index --no-build-isolation --index-url=https://mirrors.aliyun.com/pypi/simple/


FROM python:3.7.7

ENV PYTHONPATH="/srv/pkgs/:${PYTHONPATH}"
ENV PATH="/srv/pkgs/bin:${PATH}"

COPY --from=builder /srv/pkgs /srv/pkgs

RUN mkdir -p /opt/audio_detect
WORKDIR /opt/audio_detect
COPY .  /opt/audio_detect