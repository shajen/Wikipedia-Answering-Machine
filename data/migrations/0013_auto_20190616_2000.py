# Generated by Django 2.2.2 on 2019-06-16 18:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('data', '0012_auto_20190614_1717'),
    ]

    operations = [
        migrations.AlterField(
            model_name='occurrence',
            name='is_title',
            field=models.BooleanField(db_index=True, default=False),
        ),
        migrations.AlterField(
            model_name='word',
            name='is_stop_word',
            field=models.BooleanField(db_index=True, default=False),
        ),
    ]
