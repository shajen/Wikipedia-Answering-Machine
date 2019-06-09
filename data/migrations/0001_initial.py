# -*- coding: utf-8 -*-
# Generated by Django 1.11.20 on 2019-06-09 16:38
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Article',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=1024)),
                ('links', models.ManyToManyField(related_name='links_relationship', to='data.Article')),
            ],
        ),
    ]
