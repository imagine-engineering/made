#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>

static const char *find_protocol_encoding(
    Protocol *protocol,
    SEL selector,
    NSMutableSet<NSString *> *visited
) {
    NSString *protocol_name = NSStringFromProtocol(protocol);
    if ([visited containsObject:protocol_name]) {
        return NULL;
    }
    [visited addObject:protocol_name];

    for (NSUInteger required_index = 0; required_index < 2; required_index++) {
        BOOL required = required_index == 0;
        struct objc_method_description description =
            protocol_getMethodDescription(protocol, selector, required, YES);
        if (description.name != NULL && description.types != NULL) {
            return description.types;
        }
    }

    unsigned int inherited_count = 0;
    Protocol * __unsafe_unretained *inherited =
        protocol_copyProtocolList(protocol, &inherited_count);
    for (unsigned int index = 0; index < inherited_count; index++) {
        const char *encoding = find_protocol_encoding(inherited[index], selector, visited);
        if (encoding != NULL) {
            free(inherited);
            return encoding;
        }
    }
    free(inherited);
    return NULL;
}

static NSString *expected_return_nullability(NSString *class_name, NSString *selector_name) {
    if ([class_name isEqualToString:@"MTLBuffer"] &&
        [selector_name isEqualToString:@"contents"]) {
        return @"nonnull";
    }
    if ([class_name isEqualToString:@"MTLDevice"] &&
        [selector_name isEqualToString:@"name"]) {
        return @"nonnull";
    }
    BOOL nullable =
        ([class_name isEqualToString:@"MTLCommandBuffer"] &&
         ([selector_name isEqualToString:@"blitCommandEncoder"] ||
          [selector_name isEqualToString:@"computeCommandEncoder"] ||
          [selector_name isEqualToString:@"error"])) ||
        ([class_name isEqualToString:@"MTLCommandQueue"] &&
         [selector_name isEqualToString:@"commandBuffer"]) ||
        ([class_name isEqualToString:@"MTLDevice"] &&
         ([selector_name isEqualToString:@"newBufferWithLength:options:"] ||
          [selector_name isEqualToString:@"newCommandQueueWithMaxCommandBufferCount:"] ||
          [selector_name isEqualToString:@"newComputePipelineStateWithFunction:error:"] ||
          [selector_name isEqualToString:@"newLibraryWithData:error:"])) ||
        ([class_name isEqualToString:@"MTLLibrary"] &&
         [selector_name isEqualToString:@"newFunctionWithName:"]);
    return nullable ? @"nullable" : @"not-applicable";
}

static BOOL verify_row(NSDictionary *row, NSUInteger index) {
    NSString *class_name = row[@"class"];
    NSString *selector_name = row[@"selector"];
    NSString *expected_encoding = row[@"encoding"];
    NSString *return_nullability = row[@"return_nullability"];
    if (![class_name isKindOfClass:[NSString class]] || class_name.length == 0 ||
        ![selector_name isKindOfClass:[NSString class]] || selector_name.length == 0 ||
        ![expected_encoding isKindOfClass:[NSString class]] || expected_encoding.length == 0 ||
        ![return_nullability isKindOfClass:[NSString class]] || return_nullability.length == 0) {
        fprintf(stderr, "resource selector row %lu is malformed\n", (unsigned long)index);
        return NO;
    }

    NSString *reviewed_nullability =
        expected_return_nullability(class_name, selector_name);
    if (![return_nullability isEqualToString:reviewed_nullability]) {
        fprintf(stderr,
                "resource selector row %lu nullability differs for %s.%s: expected %s, observed %s\n",
                (unsigned long)index,
                class_name.UTF8String,
                selector_name.UTF8String,
                reviewed_nullability.UTF8String,
                return_nullability.UTF8String);
        return NO;
    }

    SEL selector = NSSelectorFromString(selector_name);
    const char *actual_encoding = NULL;
    if ([class_name isEqualToString:@"NSError"]) {
        Method method = class_getInstanceMethod([NSError class], selector);
        if (method != NULL) {
            actual_encoding = method_getTypeEncoding(method);
        }
    } else {
        Protocol *protocol = objc_getProtocol(class_name.UTF8String);
        if (protocol == NULL) {
            fprintf(stderr, "resource selector row %lu names an unknown protocol: %s\n",
                    (unsigned long)index, class_name.UTF8String);
            return NO;
        }
        actual_encoding = find_protocol_encoding(
            protocol,
            selector,
            [NSMutableSet set]
        );
    }

    if (actual_encoding == NULL) {
        fprintf(stderr, "resource selector row %lu is unavailable: %s.%s\n",
                (unsigned long)index, class_name.UTF8String, selector_name.UTF8String);
        return NO;
    }
    if (strcmp(actual_encoding, expected_encoding.UTF8String) != 0) {
        fprintf(stderr,
                "resource selector row %lu encoding differs for %s.%s: expected %s, observed %s\n",
                (unsigned long)index,
                class_name.UTF8String,
                selector_name.UTF8String,
                expected_encoding.UTF8String,
                actual_encoding);
        return NO;
    }
    return YES;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 2) {
            fprintf(stderr, "usage: %s EXECUTION_ABI_JSON\n", argv[0]);
            return 2;
        }

        NSError *read_error = nil;
        NSData *data = [NSData dataWithContentsOfFile:@(argv[1])
                                                options:0
                                                  error:&read_error];
        if (data == nil) {
            fprintf(stderr, "failed to read execution ABI: %s\n",
                    read_error.localizedDescription.UTF8String);
            return 1;
        }
        NSError *json_error = nil;
        id document = [NSJSONSerialization JSONObjectWithData:data options:0 error:&json_error];
        if (![document isKindOfClass:[NSDictionary class]]) {
            fprintf(stderr, "execution ABI is not a JSON object: %s\n",
                    json_error.localizedDescription.UTF8String ?: "invalid JSON");
            return 1;
        }
        NSArray *rows = ((NSDictionary *)document)[@"resource_selectors"];
        if (![rows isKindOfClass:[NSArray class]] || rows.count == 0) {
            fprintf(stderr, "execution ABI has no resource_selectors\n");
            return 1;
        }
        for (NSUInteger index = 0; index < rows.count; index++) {
            id row = rows[index];
            if (![row isKindOfClass:[NSDictionary class]] ||
                !verify_row((NSDictionary *)row, index)) {
                return 1;
            }
        }
        printf("verified %lu Metal execution selector encodings\n", (unsigned long)rows.count);
        return 0;
    }
}
