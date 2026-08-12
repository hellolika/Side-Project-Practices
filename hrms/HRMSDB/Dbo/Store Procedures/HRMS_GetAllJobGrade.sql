CREATE PROCEDURE [dbo].[HRMS_GetAllJobGrade.1.0.0]
AS
BEGIN 
	SET NOCOUNT ON
    SELECT [Id],[PositionId], [JobGradeName]
    FROM [dbo].[JobGrade]
END