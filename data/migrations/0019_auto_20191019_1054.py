# Generated by Django 2.2.2 on 2019-10-19 08:54

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('data', '0018_auto_20190703_1634'),
    ]

    operations = [
        migrations.AlterField(
            model_name='word',
            name='changed_form',
            field=models.CharField(db_index=True, max_length=100),
        ),
        migrations.AlterUniqueTogether(
            name='word',
            unique_together={('base_form', 'changed_form')},
        ),
    ]
