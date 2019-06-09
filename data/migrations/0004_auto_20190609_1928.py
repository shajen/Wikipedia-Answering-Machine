# -*- coding: utf-8 -*-
# Generated by Django 1.11.20 on 2019-06-09 19:28
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('data', '0003_occurrence'),
    ]

    operations = [
        migrations.AddField(
            model_name='occurrence',
            name='is_title',
            field=models.BooleanField(default=False),
        ),
        migrations.AlterField(
            model_name='occurrence',
            name='positions',
            field=models.TextField(),
        ),
    ]
