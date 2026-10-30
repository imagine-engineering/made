package evidence;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class GreetingTest {
    @Test
    void messageIsAvailable() {
        assertEquals("Hello from Kotlin", new Greeting().message());
    }
}
