package evidence;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

public final class GreetingTest {
    @Test
    void messageIsAvailable() {
        assertEquals("ready", Main.message());
    }
}
