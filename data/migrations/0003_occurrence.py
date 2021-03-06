# -*- coding: utf-8 -*-
# Generated by Django 1.11.20 on 2019-06-09 17:10
from __future__ import unicode_literals

import data.models
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('data', '0002_auto_20190609_1704'),
    ]

    operations = [
        migrations.CreateModel(
            name='Occurrence',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('positions', data.models.ListField(token=',')),
                ('article', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='data.Article')),
                ('word', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='data.Word')),
            ],
        ),
    ]
