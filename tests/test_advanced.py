# -*- coding: utf-8 -*-

from .context import crawler

import unittest


class AdvancedTestSuite(unittest.TestCase):
  """Advanced test cases."""
  def test_process_image(self):
    self.assertIsNone(crawler.train())


if __name__ == '__main__':
  unittest.main()
