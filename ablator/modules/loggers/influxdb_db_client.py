from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS


class InfluxDatabaseClient:
    def __init__(self, token, org, bucket) -> None:
        self._org = org
        self._bucket = bucket
        self._client = InfluxDBClient(
            url="http://localhost:8086", token=token, org=org)

    def write_data(self, data, write_option=SYNCHRONOUS) -> None:
        write_api = self._client.write_api(write_option)
        write_api.write(self._bucket, self._org, data, write_precision='s')

    def query_data(self, query):
        query_api = self._client.query_api()
        result = query_api.query(org=self._org, query=query)
        results = []
        print("result: ", result)
        for table in result:
            for record in table.records:
                results.append((record.get_value(), record.get_field()))
        print(results)
        return results
