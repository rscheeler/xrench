import contextlib
import sys

from loguru import logger


class LOGCONTROLLER:
    """Controller for Loguru logging, allowing dynamic level changes and muting/unmuting of logs
    based on module name.
    """

    def __init__(
        self,
        module_name: str,
        level: str = "INFO",
    ) -> None:
        self._handler_id: int | None = None
        self.module_name = module_name
        self._level = level.upper()

        # 1. KILL the default handler (ID 0) immediately.
        # This is why you get DEBUG logs—ID 0 is still alive and listening to everything.
        with contextlib.suppress(ValueError):
            logger.remove(0)

        # 2. Setup our custom filtered handler
        self._set_handler(self._level)

        # 3. Start in a muted state (Default)
        self.mute()

    @property
    def level(self) -> str:
        """Returns the current logging level as a string."""
        return self._level

    @level.setter
    def level(self, value: str) -> None:
        """Sets a new logging level and updates the handler accordingly."""
        self._level = value.upper()
        self._set_handler(self._level)

    def _set_handler(self, level: str) -> None:
        """Surgical replacement of the Loguru sink."""
        if self._handler_id is not None:
            with contextlib.suppress(ValueError):
                logger.remove(self._handler_id)

        # We ONLY want logs where the name starts with our module
        self._handler_id = logger.add(
            sys.stderr,
            level=level,
            filter=lambda record: record["name"].startswith(self.module_name),
        )

    def unmute(self) -> None:
        """Just opens the flow. The handler already has the correct level."""
        logger.enable(self.module_name)

    def mute(self) -> None:
        """Stops the flow at the source."""
        logger.disable(self.module_name)


# Instance
XRENCHLogger = LOGCONTROLLER(module_name="xrench")
