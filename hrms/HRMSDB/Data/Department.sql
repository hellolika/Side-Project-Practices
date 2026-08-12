BEGIN
    SET IDENTITY_INSERT [dbo].[Department] ON;
	INSERT INTO [dbo].[Department] (
        [Id],
        [DepartmentName],
        [CreatedOn],
        [CreatedBy],
        [ModifiedOn],
        [ModifiedBy]
	) VALUES
        (1, 'Unknown', GETDATE(), 1, GETDATE(), 1),
        (2, 'HR & Admin', GETDATE(), 1, GETDATE(), 1),
        (3, 'IT', GETDATE(), 1, GETDATE(), 1),
        (4, 'White Label', GETDATE(), 1, GETDATE(), 1),
        (5, 'LCG', GETDATE(), 1, GETDATE(), 1),
        (6, 'Sport', GETDATE(), 1, GETDATE(), 1),
        (7, 'Finance', GETDATE(), 1, GETDATE(), 1),
        (8, 'Customer Service', GETDATE(), 1, GETDATE(), 1)

    SET IDENTITY_INSERT [dbo].[Department] OFF;
END