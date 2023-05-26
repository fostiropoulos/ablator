from ablator.config.main import ConfigBase,configclass
@configclass
class MyOwnConfig(ConfigBase):
    originalAttri:int


def test_config_bug(tmp_path:str):
    config1=MyOwnConfig(originalAttri=10,addedAttri1=20,addedAttr2=30,add_attributes=True)
    assert config1.addedAttri1==20

if __name__=="__main__":
    tmp_dir="/tmp/dir"
    test_config_bug(tmp_dir)