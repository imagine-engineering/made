package evidence.processor;

import java.io.IOException;
import java.io.Writer;
import java.util.Set;
import javax.annotation.processing.AbstractProcessor;
import javax.annotation.processing.RoundEnvironment;
import javax.annotation.processing.SupportedAnnotationTypes;
import javax.annotation.processing.SupportedSourceVersion;
import javax.lang.model.SourceVersion;
import javax.lang.model.element.TypeElement;

@SupportedAnnotationTypes("*")
@SupportedSourceVersion(SourceVersion.RELEASE_17)
public class GreetingProcessor extends AbstractProcessor {
    private boolean generated;

    @Override
    public boolean process(Set<? extends TypeElement> annotations, RoundEnvironment round) {
        if (!generated && !round.processingOver()) {
            generated = true;
            try (Writer writer = processingEnv.getFiler().createSourceFile("evidence.GeneratedGreeting").openWriter()) {
                writer.write("package evidence; public class GeneratedGreeting { public static String message() { return \"generated\"; } }\n");
            } catch (IOException error) {
                processingEnv.getMessager().printMessage(javax.tools.Diagnostic.Kind.ERROR, error.getMessage());
            }
        }
        return false;
    }
}
