# Generated by Django 2.2.6 on 2019-12-27 15:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('data', '0025_auto_20191220_2129'),
    ]

    operations = [
        migrations.AddField(
            model_name='article',
            name='content_words',
            field=models.TextField(default=''),
        ),
        migrations.AddField(
            model_name='article',
            name='title_words',
            field=models.TextField(default=''),
        ),
    ]
