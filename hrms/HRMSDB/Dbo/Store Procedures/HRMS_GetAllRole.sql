CREATE PROCEDURE [dbo].[HRMS_GetAllRole.1.0.0]
AS
BEGIN 
	  SET NOCOUNT ON
    SELECT [Id], [RoleName],[RoleDescription]
    FROM [dbo].[RoleType]
    WHERE [IsEnable] = 1
END
