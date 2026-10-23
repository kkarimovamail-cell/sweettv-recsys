from src import config


def test_data_files_exist():
    assert config.VIEWS_FILE.is_file()
    assert config.SUBMISSION_TEMPLATE.is_file()
    for path in config.SCHEDULE_FILES.values():
        assert path.is_file()


def test_constants():
    assert config.SEED == 42
    assert config.WATCH_THRESHOLD == 0.8
    assert config.TIME_SHIFT_HOURS == 0
    assert config.K == 5


def test_get_logger_writes_file():
    logger = config.get_logger("pytest_config")
    logger.info("hello from test")
    for handler in logger.handlers:
        handler.flush()
    log_file = config.LOGS_DIR / "pytest_config.log"
    assert "hello from test" in log_file.read_text(encoding=config.ENCODING)
    # second call must not duplicate handlers
    assert len(config.get_logger("pytest_config").handlers) == 2
