# Generated by Django 2.2.2 on 2019-07-03 14:34

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('data', '0017_article_redirected_to'),
    ]

    operations = [
        migrations.AlterField(
            model_name='solution',
            name='method',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='data.Method'),
        ),
    ]
