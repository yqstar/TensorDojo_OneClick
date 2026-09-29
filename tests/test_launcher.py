"""Run: python -m unittest discover -s tests -v. Does not download dependencies."""
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import launch

class PlanTests(unittest.TestCase):
    def test_windows_cpu(self):
        packages, index = launch.install_plan('Windows', 'AMD64', (3,11))
        self.assertEqual(packages, ['torch==2.10.0']); self.assertIn('/cpu', index)
    def test_linux_cpu(self):
        self.assertIn('/cpu', launch.install_plan('Linux', 'x86_64', (3,13))[1])
    def test_mac_arm(self):
        self.assertEqual(launch.install_plan('Darwin','arm64',(3,11))[1], launch.PYPI)
    def test_mac_intel(self):
        self.assertEqual(launch.install_plan('Darwin','x86_64',(3,11))[0], ['torch==2.2.2','numpy==1.26.4'])
    def test_mac_intel_reject_313(self):
        with self.assertRaises(RuntimeError): launch.install_plan('Darwin','x86_64',(3,13))
    def test_unsupported_arch_rejected(self):
        with self.assertRaises(RuntimeError): launch.install_plan('Windows','arm64',(3,11))
    def test_future_python_install_rejected(self):
        with self.assertRaises(RuntimeError): launch.install_plan('Linux','x86_64',(3,15))
    def test_missing_offline_cache(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(launch,'ROOT',Path(tmp)):
            with self.assertRaises(RuntimeError): launch.offline_source(['torch==2.10.0'])
    def test_cache_validation_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(launch,'ROOT',Path(tmp)):
            folder=launch.wheel_directory();folder.mkdir(parents=True)
            f=folder/'test.whl';f.write_bytes(b'test fixture; not a real wheel')
            m={'platform':launch.fingerprint(),'packages':['torch==2.10.0'],
               'wheels':{f.name:launch.sha256_file(f)}}
            (folder/'manifest.json').write_text(json.dumps(m))
            self.assertIn('--no-index', launch.offline_source(m['packages']))
            f.write_bytes(b'corrupt')
            with self.assertRaises(RuntimeError): launch.offline_source(m['packages'])
    def test_reuse_existing_environment_no_pip(self):
        with patch.object(launch,'probe',return_value={'executable':'test-python','python':'3.11','torch':'2.10.0'}), \
             patch.object(launch,'run') as run:
            launch.prepare_python(offline=True)
            run.assert_not_called()
    def test_validation_stamp_skips_second_selftest(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(launch,'ROOT',Path(tmp)), \
             patch.object(launch,'run') as run:
            info={'executable':sys.executable,'python':'test','torch':'test'}
            launch.validate_once(sys.executable, info); launch.validate_once(sys.executable, info)
            self.assertEqual(run.call_count,1)
    def test_changed_modular_oracle_invalidates_validation_stamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);app=root/'app';app.mkdir()
            (app/'problems.json').write_text('[]')
            oracle=app/'rec_fixture_cases.py';oracle.write_text('VERSION = 1')
            with patch.object(launch,'ROOT',root),patch.object(launch,'APP',app),patch.object(launch,'run') as run:
                info={'executable':sys.executable,'python':'test','torch':'test'}
                launch.validate_once(sys.executable,info)
                oracle.write_text('VERSION = 2')
                launch.validate_once(sys.executable,info)
                self.assertEqual(run.call_count,2)

class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with socket.socket() as s:
            s.bind(('127.0.0.1',0));cls.port=s.getsockname()[1]
        cls.url=f'http://127.0.0.1:{cls.port}'
        cls.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        cls.log=tempfile.TemporaryFile()
        cls.process=subprocess.Popen([sys.executable,str(ROOT/'launch.py'),'--no-browser','--offline','--port',str(cls.port)],
            stdout=cls.log,stderr=subprocess.STDOUT, start_new_session=os.name!='nt')
        for _ in range(300):
            try:
                with cls.opener.open(cls.url+'/api/health',timeout=.3) as r:
                    cls.health=json.load(r)
                if cls.health['ready']:return
            except (OSError,ValueError): pass
            if cls.process.poll() is not None: break
            time.sleep(.1)
        cls.log.seek(0);out=cls.log.read().decode('utf-8',errors='replace')
        cls.tearDownClass()
        raise AssertionError('Server did not start:\n'+out)
    @classmethod
    def tearDownClass(cls):
        if cls.process.poll() is None:
            if os.name!='nt':os.killpg(cls.process.pid,signal.SIGINT)
            else:cls.process.terminate()
            try:cls.process.wait(timeout=10)
            except subprocess.TimeoutExpired:cls.process.kill();cls.process.wait()
        cls.log.close()
    def test_health_and_identity(self):
        self.assertTrue(self.health['ready']);self.assertIn('project_id',self.health)
        self.assertTrue(launch.running_here(self.port))
    def test_duplicate_launch_reuses_existing(self):
        with patch.object(launch,'prepare_python',side_effect=AssertionError('Must not install')):
            self.assertEqual(launch.main(['--no-browser','--port',str(self.port)]),0)
    def test_browser_open_callback(self):
        with patch.object(launch.webbrowser,'open',return_value=True) as opened:
            self.assertEqual(launch.main(['--port',str(self.port)]),0)
            opened.assert_called_once_with(self.url)
    def test_ui_served(self):
        with self.opener.open(self.url) as r:text=r.read().decode()
        self.assertIn('TensorDojo',text);self.assertNotIn('__LOCAL_TOKEN__',text)
    def test_real_relu_submission(self):
        with self.opener.open(self.url) as r:text=r.read().decode()
        # Token is injected into original UI as const LOCAL_TOKEN = '...'.
        import re
        token=re.search(r"LOCAL_TOKEN\s*=\s*['\"]([^'\"]+)['\"]",text).group(1)
        payload={'id':'relu','mode':'submit','code':'def relu(x):\n    return torch.relu(x)'}
        request=urllib.request.Request(self.url+'/api/run',data=json.dumps(payload).encode(),headers={
            'Content-Type':'application/json','Origin':self.url,'X-Local-Token':token})
        with self.opener.open(request,timeout=30) as r:result=json.load(r)
        self.assertEqual(result['status'],'accepted');self.assertEqual(result['passed'],12)
    def test_reject_cross_origin(self):
        request=urllib.request.Request(self.url+'/api/run',data=b'{}',headers={
            'Content-Type':'application/json','Origin':'https://example.com','X-Local-Token':'bad'})
        with self.assertRaises(urllib.error.HTTPError) as err:self.opener.open(request)
        self.assertEqual(err.exception.code,403)
    def test_recommendation_loss_metric_and_model_submissions(self):
        import re
        bank={p['id']:p for p in json.loads((ROOT/'llm_code_lab'/'problems.json').read_text(encoding='utf-8'))}
        with self.opener.open(self.url) as r:html=r.read().decode()
        token=re.search(r"LOCAL_TOKEN\s*=\s*['\"]([^'\"]+)['\"]",html).group(1)
        for pid,mode in (('bce_with_logits','submit'),('binary_auc','submit'),('mmoe','run')):
            with self.subTest(pid=pid):
                payload={'id':pid,'code':bank[pid]['solution'],'mode':mode}
                request=urllib.request.Request(self.url+'/api/run',data=json.dumps(payload).encode(),headers={
                    'Content-Type':'application/json','Origin':self.url,'X-Local-Token':token})
                with self.opener.open(request,timeout=30) as r:result=json.load(r)
                self.assertEqual(result['status'],'accepted',result)
                self.assertEqual(result['mode'],mode)
                if mode=='run':self.assertEqual(result['total'],2)
                if pid=='binary_auc':self.assertTrue(all(c['checks']['gradient'] is None for c in result['cases']))
    def test_occupied_foreign_port(self):
        with socket.socket() as s:
            s.bind(('127.0.0.1',0));s.listen();port=s.getsockname()[1]
            with self.assertRaises(RuntimeError):launch.running_here(port)

    def test_llm_and_rl_submissions_through_local_service(self):
        import re
        bank = {p['id']: p for p in json.loads((ROOT/'llm_code_lab'/'problems.json').read_text(encoding='utf-8'))}
        with self.opener.open(self.url) as r:html = r.read().decode()
        token = re.search(r"LOCAL_TOKEN\s*=\s*['\"]([^'\"]+)['\"]", html).group(1)
        for pid, mode in (('lora_linear', 'submit'), ('causal_lm_loss', 'submit'),
                          ('gae_advantages', 'submit'), ('ppo_clipped_loss', 'submit'),
                          ('grpo_loss', 'run')):
            with self.subTest(pid=pid):
                payload = {'id': pid, 'code': bank[pid]['solution'], 'mode': mode}
                request = urllib.request.Request(self.url+'/api/run', data=json.dumps(payload).encode(), headers={
                    'Content-Type': 'application/json', 'Origin': self.url, 'X-Local-Token': token})
                with self.opener.open(request, timeout=30) as r:result = json.load(r)
                self.assertEqual(result['status'], 'accepted', result)
                self.assertEqual(result['mode'], mode)
                if mode == 'run':self.assertEqual(result['total'], 2)
                else:self.assertGreaterEqual(result['total'], 12)
                if pid == 'gae_advantages':
                    self.assertTrue(all(c['checks']['gradient'] is None for c in result['cases']))

if __name__=='__main__':unittest.main()
