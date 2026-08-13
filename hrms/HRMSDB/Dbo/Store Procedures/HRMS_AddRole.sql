CREATE PROCEDURE [dbo].[HRMS_AddRole.1.0.0]
    @roleName [nvarchar](50),
    @roleDescription [nvarchar](200),
    @createdBy INT
AS
BEGIN
    SET NOCOUNT ON
    IF EXISTS(SELECT TOP 1 1 FROM [dbo].[RoleType] WHERE [RoleName] COLLATE SQL_Latin1_General_CP1_CS_AS = @roleName COLLATE SQL_Latin1_General_CP1_CS_AS) 
    BEGIN
        SELECT 38 AS ErrorCode;
        RETURN;
    END

    INSERT INTO [dbo].[RoleType]
    ([RoleName], [RoleDescription],[IsEnable],[CreatedBy], [CreatedOn], [ModifiedOn])
    VALUES(@roleName, @roleDescription,1, @createdBy, GETDATE(), GETDATE());

    SELECT 0 AS ErrorCode;

END