# Generated by Django 3.0.2 on 2020-01-05 22:25

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('data', '0027_auto_20191227_2039'),
    ]

    operations = [
        migrations.AddField(
            model_name='method',
            name='is_enabled',
            field=models.BooleanField(db_index=True, default=True),
        ),
    ]