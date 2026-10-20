"""Local endpoint selection and request budgets; no model weights/API calls."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ..medqa import cli
from ..medqa.model import CompatibleChat


class LocalModelTest(unittest.TestCase):
    def args(self,**kw):
        values={'model_name':'local-qwen','model_base_url':'http://127.0.0.1:8010/v1',
                'model_api_key_env':None,'model_timeout_seconds':180}
        return SimpleNamespace(**{**values,**kw})

    def test_explicit_local_endpoint_never_uses_cloud_credentials(self):
        with patch.dict(os.environ,{'DEEPSEEK_MODEL':'cloud','DEEPSEEK_BASE_URL':'https://api.deepseek.com',
                'DEEPSEEK_API_KEY':'cloud-secret','MEDQA_API_KEY':'other-cloud-secret'},clear=True):
            model,base,key,timeout=cli.model_connection(self.args())
        self.assertEqual((model,base,key,timeout),('local-qwen','http://127.0.0.1:8010/v1','local-moreagent',180))

    def test_local_missing_model_does_not_inherit_deepseek_model(self):
        with patch.dict(os.environ,{'DEEPSEEK_MODEL':'cloud','DEEPSEEK_API_KEY':'cloud-secret'},clear=True):
            with self.assertRaises(cli.ConfigurationError):cli.model_connection(self.args(model_name=None))

    def test_remote_environment_fallback_is_preserved(self):
        with patch.dict(os.environ,{'DEEPSEEK_MODEL':'cloud','DEEPSEEK_BASE_URL':'https://api.deepseek.com',
                'DEEPSEEK_API_KEY':'secret'},clear=True):
            self.assertEqual(cli.model_connection(self.args(model_name=None,model_base_url=None))[:3],
                             ('cloud','https://api.deepseek.com','secret'))

    def test_key_environment_override_and_invalid_timeout(self):
        with patch.dict(os.environ,{'LOCAL_KEY':'custom'},clear=True):
            self.assertEqual(cli.model_connection(self.args(model_api_key_env='LOCAL_KEY'))[2],'custom')
            with self.assertRaises(cli.ConfigurationError):cli.model_connection(self.args(model_timeout_seconds=float('nan')))
            with self.assertRaises(cli.ConfigurationError):cli.model_connection(self.args(model_base_url='http:invalid'))

    def test_long_local_request_is_bounded_by_remaining_question_time(self):
        sent=[]
        def create(**kw):
            sent.append(kw)
            return SimpleNamespace(usage=None,choices=[SimpleNamespace(
                message=SimpleNamespace(content='{}'),finish_reason='stop')])
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        chat=CompatibleChat(model='qwen',base_url='http://127.0.0.1:8010/v1',api_key='local',
            max_tokens=1800,timeout_seconds=180,client=client)
        from langchain_core.messages import HumanMessage
        chat.invoke_bounded([HumanMessage(content='fixture')],remaining_seconds=120,remaining_tokens=500)
        self.assertEqual(sent[0]['timeout'],120)
        self.assertEqual(sent[0]['max_tokens'],500)
        self.assertNotIn('extra_body',sent[0])

    def test_qwen3_local_disables_thinking_preserves_usage_and_does_not_change_cloud(self):
        sent=[]
        usage={'prompt_tokens':101,'completion_tokens':17,'total_tokens':118}
        def create(**kw):
            sent.append(kw)
            return SimpleNamespace(usage=SimpleNamespace(model_dump=lambda:usage),model='qwen3-14b-bf16',
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"answer":"A"}'),finish_reason='stop')])
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        from langchain_core.messages import HumanMessage
        for base in ('http://127.0.0.1:8010/v1','https://example.org/v1'):
            chat=CompatibleChat(model='qwen3-14b-bf16',base_url=base,api_key='local',max_tokens=900,client=client)
            reply=chat.invoke([HumanMessage(content='fixture')])
            self.assertEqual(reply.response_metadata['token_usage'],usage)
        self.assertEqual(sent[0]['extra_body'],{'chat_template_kwargs':{'enable_thinking':False}})
        self.assertNotIn('extra_body',sent[1])


if __name__=='__main__':unittest.main()
