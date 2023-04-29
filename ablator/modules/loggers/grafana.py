import os
import json
import base64
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from typing import Any
import plotly.graph_objs as go

from grafanalib.core import *
from omegaconf import OmegaConf
from grafanalib._gen import DashboardEncoder
from influxdb_db_client import InfluxDatabaseClient

from influxdb_client import Point
from ablator.config.main import ConfigBase
from ablator.config.utils import flatten_nested_dict


class GrafanaLogger():
    def __init__(self, grafana_token, influxdb_token, influxdb_bucket, influxdb_org) -> None:
        self.grafana_server: str = "localhost:3000"
        self.grafana_api_key: str = grafana_token
        self.influxdb_bucket: str = influxdb_bucket
        self.influxdb_client: InfluxDatabaseClient = InfluxDatabaseClient(influxdb_token, influxdb_org, influxdb_bucket)

    def _get_dashboard_json(self, dashboard, overwrite=False, message="Updated by grafanlib") -> str:
        return json.dumps(
            {
                "dashboard": dashboard.to_json_data(),
                "overwrite": overwrite,
                "message": message
            }, sort_keys=True, indent=2, cls=DashboardEncoder)

    def upload_to_grafana(self, dashboard, verify=True):
        json = self._get_dashboard_json(dashboard, overwrite=True)
        headers: dict[str, str] = {'Authorization': f"Bearer {self.grafana_api_key}",
                                   'Content-Type': 'application/json'}
        try:
            r = requests.post(f"http://{self.grafana_server}/api/dashboards/db",
                              data=json, headers=headers, verify=verify)
            print(f"{r.status_code} - {r.content}")
            return r.json().get('uid')
        except Exception as e:
            print(f"Connection refused by the Grafana server: {str(e)}")

    def generate_dashboard(self, dashboard_title, description, panels) -> Dashboard:
        return Dashboard(title=dashboard_title,
                         description=description,
                         tags=['ablator'],
                         timezone="browser",
                         panels=panels).auto_panel_ids()

    def add_table(self, k, v: pd.DataFrame, itr):
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(v.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=v.transpose().values.tolist(),
                       fill_color='lavender',
                       align='left'))
        ])
        return Text(title=str(k),
                    description=f"iteration_{itr}",
                    gridPos=GridPos(h=10, w=6.5, x=10, y=10),
                    content=fig.to_html(),
                    mode='html')

    def add_image(self, k, v, itr) -> Text:
        buffered = BytesIO()
        v.save(buffered, format="JPEG")
        data = base64.b64encode(buffered.getvalue())

        encoded_img = str(data, "utf-8")
        imageData: str = "data:image/jpeg;base64, " + encoded_img

        p: Point = Point("Ablator_Images").tag(
            "ablator", "Images").field("iteration", itr).field("image", imageData)
        self.influxdb_client.write_data(p)

    def add_image_panel(self, k, v: Image, itr):
        self.add_image(k, v, itr)
        query_expr = f'from(bucket: "{self.influxdb_bucket}") |> range(start: 0)|> filter(fn: (r) => r.ablator == "Images")|> filter(fn: (r) => r._field == "image")'
        return Table(title='My Panel',
                     description=f"iteration_{itr}",
                     gridPos=GridPos(h=10, w=10, x=0, y=0),
                     dataSource='Ablator-InfluxDB',
                     targets=[Target(expr=query_expr)])

    def add_text(self, k, v, itr) -> Text:
        return Text(title=str(k), description=f"iteration_{itr}", gridPos=GridPos(h=10, w=10, x=0, y=10), content=v, mode='markdown')

    def write_config(self, config: ConfigBase) -> Text:
        hparams: dict[str, Any] = flatten_nested_dict(config.to_dict())
        run_config: str = OmegaConf.to_yaml(OmegaConf.create(hparams)).replace("\n", "\n\n")
        return Text(title="config", description=0, gridPos=GridPos(h=12, w=10, x=6.5, y=9), content=run_config, mode='markdown')

    # This function adds scalars to the existing metrics in InfluxDB
    def add_scalar(self, k, v, itr) -> None:
        p: Point = Point(f"Ablator_{k}").tag("ablator", k).field(
            "iteration", itr).field(k, v)
        self.influxdb_client.write_data(p)

    def add_scalars(self, k, v, itr):
        data_points = []
        for _k, _v in v.items():
            p: Point = Point(f"Ablator_{_k}").tag("ablator", _k).field(
                "iteration", itr).field(_k, _v)
            data_points.append(p)
        self.influxdb_client.write_data(data_points)

    def add_time_series_panel(self, k, v, itr) -> TimeSeries:
        self.add_scalar(k, v, itr)
        query_expr = f'from(bucket: "{self.influxdb_bucket}")|> range(start: 0)|> filter(fn: (r) => r.ablator == "{k}")|> filter(fn: (r) => r._field == "{k}")'

        return TimeSeries(title=str(k),
                          description=f"iteration_{itr}",
                          gridPos=GridPos(h=10, w=10, x=0, y=0),
                          dataSource='Ablator-InfluxDB',
                          targets=[Target(expr=query_expr)]
                          )

    def add_multiline_time_series_panel(self, k, v: dict[str, float | int], itr) -> TimeSeries:
        targets = []
        self.add_scalars(k, v, itr)
        for _k, _v in v.items():
            query_expr = f'from(bucket: "{self.influxdb_bucket}")|> range(start: 0)|> filter(fn: (r) => r.ablator == "{_k}")|> filter(fn: (r) => r._field == "{_k}")'
            targets.append(Target(expr=query_expr))

        return TimeSeries(title=str(k),
                          description=f"iteration_{itr}",
                          gridPos=GridPos(h=10, w=10, x=0, y=0),
                          dataSource='Ablator-InfluxDB',
                          targets=targets
                          )
