# -*- coding: utf-8 -*-
# Generated by Django 1.11.20 on 2019-06-09 19:56
from __future__ import unicode_literals

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('data', '0005_auto_20190609_1954'),
    ]

    operations = [
        migrations.RenameField(
            model_name='word',
            old_name='original_form',
            new_name='changed_form',
        ),
    ]
