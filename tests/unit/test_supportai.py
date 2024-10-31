""" SupportAI Unit Tests """
import unittest
from unittest import mock

from support_ai import ai_bot

# pylint: disable=no-self-use


class TestSupportAI(unittest.TestCase):
    """ Unit Tests for Support AI. """

    @mock.patch('support_ai.ai_bot.input')
    @mock.patch.object(ai_bot, 'parse_args')
    @mock.patch.object(ai_bot, 'get_config')
    @mock.patch.object(ai_bot, 'Chain')
    def test_ai_bot_quit(self, _mock_chain, _mock_get_config, _mock_parse_args,
                         mock_input):
        """
        Test api bot quit.
        """
        mock_input.return_value = 'quit'
        ai_bot.main()
