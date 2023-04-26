import os
import json
import random
import socket

import requests
import pandas as pd
from typing import Any
from PIL import Image
from sys import platform
from bs4 import BeautifulSoup
from omegaconf import OmegaConf


from grafanalib.core import *
import plotly.graph_objs as go
from grafanalib._gen import DashboardEncoder
from prometheus_db_client import PrometheusClient
from ablator.config.main import ConfigBase
from ablator.config.utils import flatten_nested_dict


class GrafanaLogger():
    def __init__(self) -> None:
        self.grafana_server: str = "localhost:3000"
        self.grafana_api_key: str = self._get_grafana_api_key()
        self.dashboard_uid: str = self._get_uid()
        self.grafana_img_folder: str | None = self._find_grafana_img_folder()
        self.ip_address: str = self._get_ip_address()
        self.pushgateway_server: str = f'http://{self.ip_address}:9091'
        self.prometheus_client = PrometheusClient(self.pushgateway_server)
        self._create_prometheus_data_source()

    def _find_grafana_img_folder(self):
        search_dir = ""
        # TODO: add support for other machine and test it
        if platform == "linux" or platform == "linux2":
            search_dir = '/etc'
        elif platform == "darwin":
            search_dir = '/opt'
        elif platform == "win32":
            search_dir = 'C://'

        for root, dirs, files in os.walk(search_dir):
            for name in files:
                if name == "grafana_icon.svg":
                    return root

    def _get_uid(self) -> str:
        dashboard_uid: str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=7))
        return dashboard_uid

    def _get_ip_address(self):
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        return ip_address

    def _get_grafana_api_key(self):
        org_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=7))
        # First create an org
        # Should get username and password from user ideally admin admin are garfana defaults
        url = f'http://admin:admin@{self.grafana_server}/api'
        data = {"name": org_name}
        headers = {'Content-Type': 'application/json'}
        r = requests.post(f"{url}/orgs", headers=headers, json=data)
        if r.status_code == 200:
            # change to this org
            orgId = r.json().get('orgId')
            requests.post(f"{url}/user/using/{orgId}")
            # get the api key form grafana
            r = requests.post(f"{url}/auth/keys", headers=headers, json={"name": "apikeycurl", "role": "Admin"})
            return r.json().get('key')

    def _create_prometheus_data_source(self):
        url = f'http://{self.grafana_server}/api/datasources'
        data = {"name": "Prometheus", "type": "prometheus",
                "url": f"http://{self.ip_address}:9090/", "access": "proxy", "basicAuth": False}
        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.grafana_api_key}'}
        requests.post(url, headers=headers, json=data)

    def _get_dashboard_json(self, dashboard, overwrite=False, message="Updated by grafanlib") -> str:
        return json.dumps(
            {
                "dashboard": dashboard.to_json_data(),
                "overwrite": overwrite,
                "uid": self.dashboard_uid,
                "message": message
            }, sort_keys=True, indent=2, cls=DashboardEncoder)

    def upload_to_grafana(self, dashboard, verify=True) -> None:
        json = self._get_dashboard_json(dashboard, overwrite=True)
        headers: dict[str, str] = {'Authorization': f"Bearer {self.grafana_api_key}",
                                   'Content-Type': 'application/json'}
        try:
            r = requests.post(f"http://{self.grafana_server}/api/dashboards/db",
                              data=json, headers=headers, verify=verify)
            print(f"{r.status_code} - {r.content}")
        except Exception as e:
            print(f"Connection refused by the Grafana server: {str(e)}")

    def generate_dashboard(self, dashboard_title, description, panels) -> Dashboard:
        return Dashboard(title=dashboard_title,
                         description=description,
                         tags=['ablator'],
                         timezone="browser",
                         panels=panels).auto_panel_ids()

    def add_table(self, k, v: pd.DataFrame, itr) -> Text:
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

    def add_image(self, k, v: Image, itr) -> Text:
        path = os.path.join(self.grafana_img_folder, "ablator_log_images")

        if not os.path.exists(path):
            os.makedirs(path)

        v.save(f"{path}/{itr}.jpeg")

        # get the file count to update the html file
        _, _, files = next(os.walk(path))
        file_count = len(files)

        # read the html file
        data: str = ""
        with open("image-slider.html", encoding="utf-8") as f:
            data: str = f.read()
        # update the total number of files inside the html files using beautiful soup
        soup = BeautifulSoup(data, 'html.parser')
        soup.find(id='total_images').string = str(file_count)
        soup.find(id='valR')['max'] = str(file_count-1)
        html = soup.prettify("utf-8")

        # save the updated html file back again
        with open("image-slider.html", "wb") as file:
            file.write(html)

        # create a text panel in grafana to add the images
        return Text(title=str(k),
                    description=f"iteration_{itr}",
                    gridPos=GridPos(h=10, w=6.5, x=10, y=0),
                    content=str(data),
                    mode='html')

    def add_text(self, k, v, itr) -> Text:
        return Text(title=str(k), description=f"iteration_{itr}", gridPos=GridPos(h=10, w=10, x=0, y=10), content=v, mode='markdown')

    def write_config(self, config: ConfigBase) -> Text:
        hparams: dict[str, Any] = flatten_nested_dict(config.to_dict())
        run_config: str = OmegaConf.to_yaml(OmegaConf.create(hparams)).replace("\n", "\n\n")
        return Text(title="Config", gridPos=GridPos(h=10, w=10, x=0, y=10), content=run_config, mode='markdown')

    # This function adds scalars to the existing metrics in prometheus
    def add_scalar(self, k, v, itr) -> TimeSeries:
        self.prometheus_client.push_to_prometheus(k, v, itr)

    def add_time_series_panel(self, k, v, itr) -> TimeSeries:
        query_expr: str = f"{k}"
        self.add_scalar(k, v, itr)

        return TimeSeries(title=str(k),
                          description=f"iteration_{itr}",
                          gridPos=GridPos(h=10, w=10, x=0, y=0),
                          dataSource='Prometheus',
                          targets=[Target(expr=query_expr),]
                          )

    def add_multiline_time_series_panel(self, k, v: dict[str, float | int], itr) -> None:
        targets = []
        self.prometheus_client.push_metrics(v, itr)
        for _k, _v in v.items():
            query_expr: str = f"{_k}"
            targets.append(Target(expr=query_expr))

        return TimeSeries(title=str(k),
                          description=f"iteration_{itr}",
                          gridPos=GridPos(h=10, w=10, x=0, y=0),
                          dataSource='Prometheus',
                          targets=targets
                          )
